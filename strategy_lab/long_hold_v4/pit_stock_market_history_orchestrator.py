"""Run isolated BaoStock collection shards and rebuild one governed observation bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .pit_stock_market_history_builder import (
    MAX_SAFE_COLLECTION_WORKERS,
    RAW_DIR,
    ROOT,
    _atomic_json,
    _relative,
    _sha256,
    build_outputs,
    resolve_public_source_ip,
)


OUTPUT_PATH = RAW_DIR / "orchestration_latest.json"
LOG_DIR = ROOT / "logs" / "long_hold_v4" / "stock_market_history_collection"


def worker_command(
    as_of: str,
    server_ip: str,
    workers: int,
    shard_index: int,
    collect_limit: int | None,
    sleep_seconds: float,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-X",
        "utf8",
        "-m",
        "strategy_lab.long_hold_v4.pit_stock_market_history_builder",
        "--as-of",
        as_of,
        "--server-ip",
        server_ip,
        "--collect-only",
        "--shard-count",
        str(workers),
        "--shard-index",
        str(shard_index),
        "--sleep-seconds",
        str(sleep_seconds),
    ]
    if collect_limit is not None:
        command.extend(["--collect-limit", str(collect_limit)])
    return command


def _read_fresh_shard_report(path: Path, started_at: pd.Timestamp) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, "shard_report_missing"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        created_at = pd.Timestamp(report["created_at"])
    except (OSError, UnicodeError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return None, f"shard_report_unreadable:{type(exc).__name__}"
    if created_at.tzinfo is not None:
        created_at = created_at.tz_localize(None)
    if created_at < started_at.tz_localize(None):
        return None, "shard_report_stale"
    return report, "ok"


def run_orchestration(
    as_of: str,
    workers: int,
    collect_limit: int | None,
    server_ip: str | None = None,
    sleep_seconds: float = 0.05,
    worker_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_SAFE_COLLECTION_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_SAFE_COLLECTION_WORKERS}")
    if collect_limit is not None and collect_limit < 0:
        raise ValueError("collect_limit must be non-negative")
    as_of_date = pd.Timestamp(as_of).normalize().date().isoformat()
    source_ip = server_ip or resolve_public_source_ip()
    started_at = pd.Timestamp.now(tz="Asia/Shanghai")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    processes: list[tuple[int, subprocess.Popen[str], list[str]]] = []
    try:
        for shard_index in range(workers):
            command = worker_command(
                as_of=as_of_date,
                server_ip=source_ip,
                workers=workers,
                shard_index=shard_index,
                collect_limit=collect_limit,
                sleep_seconds=sleep_seconds,
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            processes.append((shard_index, process, command))
    except Exception:
        for _, process, _ in processes:
            process.terminate()
            process.communicate()
        raise

    shard_results: list[dict[str, Any]] = []
    for shard_index, process, command in processes:
        timed_out = False
        try:
            output, _ = process.communicate(timeout=worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
        log_path = LOG_DIR / f"shard_{shard_index}_of_{workers}.log"
        log_path.write_text(output, encoding="utf-8")
        shard_report_path = RAW_DIR / f"collection_status_shard_{shard_index}_of_{workers}.json"
        shard_report, report_status = _read_fresh_shard_report(shard_report_path, started_at)
        shard_results.append(
            {
                "shard_index": shard_index,
                "pid": process.pid,
                "return_code": process.returncode,
                "timed_out": timed_out,
                "command": command,
                "log_path": _relative(log_path),
                "log_sha256": _sha256(log_path),
                "shard_report_status": report_status,
                "shard_report": shard_report,
            }
        )

    build = build_outputs(as_of_date)
    workers_pass = all(
        item["return_code"] == 0
        and not item["timed_out"]
        and item["shard_report_status"] == "ok"
        and not item["shard_report"].get("failed_assets", [])
        and not item["shard_report"].get("deferred_assets", [])
        for item in shard_results
    )
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_at": started_at.isoformat(),
        "as_of_date": as_of_date,
        "workers": workers,
        "global_collect_limit": collect_limit,
        "source_server_ip": source_ip,
        "worker_timeout_seconds": worker_timeout_seconds,
        "worker_collection_pass": workers_pass,
        "shards": shard_results,
        "build": {
            "qualification_status": build["qualification_status"],
            "completed_assets": build["completed_assets"],
            "trade_completed_assets": build["trade_completed_assets"],
            "valuation_completed_assets": build["valuation_completed_assets"],
            "target_assets": build["target_assets"],
            "missing_asset_count": build["missing_asset_count"],
            "failed_asset_count": build["failed_asset_count"],
            "trade_rows": build["trade_state"]["rows"],
            "valuation_rows": build["valuation"]["rows"],
            "trade_state_qualification_status": build["trade_state"]["qualification_status"],
            "valuation_qualification_status": build["valuation"]["qualification_status"],
            "trade_state_historical_backtest_allowed": build["trade_state"]["historical_backtest_allowed"],
            "valuation_historical_backtest_allowed": build["valuation"]["historical_backtest_allowed"],
            "historical_backtest_allowed": build["historical_backtest_allowed"],
        },
        "code": [
            {"path": _relative(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            {
                "path": _relative(ROOT / "strategy_lab" / "long_hold_v4" / "pit_stock_market_history_builder.py"),
                "sha256": _sha256(
                    ROOT / "strategy_lab" / "long_hold_v4" / "pit_stock_market_history_builder.py"
                ),
            },
        ],
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(payload, OUTPUT_PATH)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--workers", type=int, default=1)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--collect-limit", type=int)
    scope.add_argument("--all-assets", action="store_true")
    parser.add_argument("--server-ip")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--worker-timeout-seconds", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_orchestration(
        as_of=args.as_of,
        workers=args.workers,
        collect_limit=None if args.all_assets else args.collect_limit,
        server_ip=args.server_ip,
        sleep_seconds=args.sleep_seconds,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "worker_collection_pass": result["worker_collection_pass"],
                "workers": result["workers"],
                "global_collect_limit": result["global_collect_limit"],
                **result["build"],
                "orchestration_report": _relative(OUTPUT_PATH),
            },
            ensure_ascii=False,
        )
    )
    if not result["worker_collection_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
